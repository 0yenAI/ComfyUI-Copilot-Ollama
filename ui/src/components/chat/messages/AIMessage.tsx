// Copyright (C) 2025 AIDC-AI
// Licensed under the MIT License.

import { MemoizedReactMarkdown } from "../../markdown";
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeExternalLinks from 'rehype-external-links';
import { BaseMessage } from './BaseMessage';
import { ChatResponse } from "../../../types/types";
import { useRef, useState } from "react";
interface AIMessageProps {
  content: string;
  name?: string;
  avatar: string;
  format?: string;
  onOptionClick?: (option: string) => void;
  extComponent?: React.ReactNode;
}

export function AIMessage({ content, name = 'Assistant', format, onOptionClick, extComponent }: AIMessageProps) {
  const markdownWrapper = useRef<HTMLDivElement | null>()
  const renderContent = () => {
    try {
      const response = JSON.parse(content) as ChatResponse;
      const guides = response.ext?.find(item => item.type === 'guides')?.data || [];

      return (
        <div className="space-y-3">
          {format === 'markdown' && response.text ? (
            <div ref={markdownWrapper as React.RefObject<HTMLDivElement>}>
              <MemoizedReactMarkdown
                rehypePlugins={[
                  [rehypeExternalLinks, { target: '_blank' }],
                  rehypeKatex
                ]}
                remarkPlugins={[remarkGfm, remarkMath]}
                className="prose prose-xs prose-neutral prose-a:text-accent-foreground/50 break-words [&>*]:!my-1 leading-relaxed text-xs
                           prose-headings:font-semibold
                           prose-h1:text-base
                           prose-h2:text-sm
                           prose-h3:text-xs
                           prose-h4:text-xs
                           prose-p:text-xs
                           prose-ul:text-xs
                           prose-ol:text-xs
                           prose-li:text-xs
                           prose-code:text-xs
                           prose-pre:text-xs"
                components={{
                  p: ({ children }) => {
                    return <p className="!my-0.5 leading-relaxed text-xs">{children}</p>
                  },
                  h1: ({ children }) => {
                    return <h1 className="text-base font-semibold !my-1">{children}</h1>
                  },
                  h2: ({ children }) => {
                    return <h2 className="text-sm font-semibold !my-1">{children}</h2>
                  },
                  h3: ({ children }) => {
                    return <h3 className="text-xs font-semibold !my-1">{children}</h3>
                  },
                  h4: ({ children }) => {
                    return <h4 className="text-xs font-semibold !my-1">{children}</h4>
                  },
                  table: ({ children }) => (
                    <table className="border-solid border border-[#979797] w-[100%] text-xs">{children}</table>
                  ),
                  th: ({ children }) => (
                    <th className="border-solid bg-[#E5E7ED] dark:bg-[#FFFFFF] dark:text-[#000000] border border-[#979797] text-center pt-2 text-xs">{children}</th>
                  ),
                  td: ({ children }) => {
                    if (Array.isArray(children) && children?.length > 0) {
                      const list: any[] = [];
                      const length = children.length;
                      for (let i = 0; i < length; i++) {
                        if (children[i] === '<br>') {
                          list.push(<br />)
                        } else {
                          list.push(children[i])
                        }
                      }
                      children = list;
                    }
                    return (
                      <td className="border-solid border border-[#979797] text-center text-xs">{children}</td>
                    )
                  },
                  code: ({ children }) => {
                    const [copied, setCopied] = useState(false);
                    
                    const handleCopy = async () => {
                      try {
                        await navigator.clipboard.writeText(children as string);
                        setCopied(true);
                        setTimeout(() => setCopied(false), 2000);
                      } catch (err) {
                        console.error('Failed to copy text:', err);
                      }
                    };
                    
                    return (
                      <div className="relative group">
                        <code className="text-xs bg-gray-100 text-gray-800 rounded px-1">{children}</code>
                        <button 
                          onClick={handleCopy}
                          className="absolute top-0 right-0 bg-gray-200 rounded p-1 opacity-0 group-hover:opacity-100 transition-opacity hover:bg-gray-300 z-10"
                          aria-label="Copy code"
                        >
                          {copied ? (
                            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                            </svg>
                          ) : (
                            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                              <path d="M8 2a1 1 0 000 2h2a1 1 0 100-2H8z" />
                              <path d="M3 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v6h-4.586l1.293-1.293a1 1 0 00-1.414-1.414l-3 3a1 1 0 000 1.414l3 3a1 1 0 001.414-1.414L10.414 13H15v3a2 2 0 01-2 2H5a2 2 0 01-2-2V5z" />
                            </svg>
                          )}
                        </button>
                      </div>
                    );
                  },
                  pre: ({ children }) => {
                    return <pre className="text-xs bg-gray-100 text-gray-800 rounded p-2 overflow-x-auto">{children}</pre>
                  },
                  img: ({ node, ...props }) => (
                    <div className="w-1/2 mx-auto">
                      <img
                        {...props}
                        loading="lazy"
                        className="w-full h-auto block" 
                        onError={(e) => {
                          console.warn('Image failed to load:', props.src, 'Error:', e);
                          e.currentTarget.style.opacity = '0';
                        }}
                      />
                    </div>
                  ),
                }}
              >
                {response.text}
              </MemoizedReactMarkdown>
            </div>
          ) : response.text ? (
            <p className="whitespace-pre-wrap text-left">
              {response.text}
            </p>
          ) : null}

          {guides.length > 0 && (
            <div className="flex flex-wrap gap-2 mt-2">
              {guides.map((guide: string, index: number) => (
                <button
                  key={index}
                  className="px-3 py-1.5 bg-white text-gray-700 rounded-md hover:bg-gray-50 transition-colors text-[12px] w-[calc(50%-0.25rem)]"
                  onClick={() => onOptionClick?.(guide)}
                >
                  {guide}
                </button>
              ))}
            </div>
          )}

          {extComponent}
        </div>
      );
    } catch {
      return <p className="whitespace-pre-wrap text-left">{content}</p>;
    }
  };

  return (
    <BaseMessage name={name}>
      <div className="w-full rounded-lg bg-gray-50 p-4 text-gray-700 text-sm break-words overflow-hidden">
        {renderContent()}
      </div>
    </BaseMessage>
  );
} 